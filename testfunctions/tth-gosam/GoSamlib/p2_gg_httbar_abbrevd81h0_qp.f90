module     p2_gg_httbar_abbrevd81h0_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(52), public :: abb81
   complex(ki), public :: R2d81
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_color_qp, only: TR
      use p2_gg_httbar_globalsl1_qp, only: epspow
      implicit none
      abb81(1)=1.0_ki/(-mT**2+es34)
      abb81(2)=sqrt(mT**2)
      abb81(3)=NC**(-1)
      abb81(4)=spbl4k2**(-1)
      abb81(5)=spbl5k2**(-1)
      abb81(6)=spak2l3**(-1)
      abb81(7)=spbl3k2**(-1)
      abb81(8)=abb81(2)**3
      abb81(9)=i_*TR*e*gHT*abb81(1)*gs**4
      abb81(10)=abb81(9)*mT
      abb81(11)=abb81(8)*abb81(10)
      abb81(12)=abb81(9)*abb81(2)**4
      abb81(11)=abb81(11)+abb81(12)
      abb81(12)=c2*abb81(3)
      abb81(12)=abb81(12)-c3
      abb81(11)=-abb81(11)*abb81(12)
      abb81(13)=spae1l4*abb81(11)
      abb81(14)=abb81(8)*abb81(9)
      abb81(15)=abb81(9)*abb81(2)**2
      abb81(16)=abb81(15)*mT
      abb81(14)=abb81(16)+abb81(14)
      abb81(16)=-mT*abb81(12)
      abb81(14)=-abb81(14)*abb81(16)
      abb81(17)=abb81(14)*abb81(4)
      abb81(18)=spae1k1*spbk2k1
      abb81(19)=abb81(18)*abb81(17)
      abb81(20)=-abb81(15)*abb81(12)
      abb81(21)=abb81(20)*spal3l4
      abb81(22)=spbl3k1*spae1k1
      abb81(23)=-abb81(21)*abb81(22)
      abb81(13)=abb81(23)+abb81(13)+abb81(19)
      abb81(19)=spae2l5*spbe2e1
      abb81(13)=abb81(19)*abb81(13)
      abb81(23)=mT*abb81(2)
      abb81(24)=abb81(23)*abb81(9)
      abb81(15)=abb81(15)+abb81(24)
      abb81(15)=-abb81(15)*abb81(12)
      abb81(25)=abb81(15)*spae1l5
      abb81(26)=abb81(25)*spbe2e1
      abb81(27)=abb81(26)*spae2k2
      abb81(28)=spbk2k1*abb81(27)
      abb81(29)=spae1e2*spbk2e2
      abb81(30)=abb81(5)*abb81(29)
      abb81(31)=abb81(30)*abb81(14)
      abb81(32)=spbk1e1*abb81(31)
      abb81(28)=abb81(32)+abb81(28)
      abb81(28)=spak1l4*abb81(28)
      abb81(32)=-spae1l3*spae2l5
      abb81(33)=spae2l3*spae1l5
      abb81(32)=abb81(33)+abb81(32)
      abb81(33)=spbl3k2*abb81(4)
      abb81(34)=abb81(33)*spbe2e1
      abb81(32)=abb81(34)*abb81(32)
      abb81(35)=spbl3e2*spae1e2
      abb81(36)=spal3l4*abb81(5)
      abb81(37)=-spbk2e1*abb81(36)*abb81(35)
      abb81(32)=abb81(37)+abb81(32)
      abb81(37)=-abb81(9)*abb81(12)
      abb81(8)=-abb81(37)*abb81(8)*mT
      abb81(32)=abb81(8)*abb81(32)
      abb81(8)=abb81(8)*abb81(36)*abb81(29)
      abb81(38)=abb81(29)*spak2l5
      abb81(39)=-abb81(21)*abb81(38)
      abb81(8)=abb81(8)+abb81(39)
      abb81(8)=spbl3e1*abb81(8)
      abb81(39)=-abb81(19)*abb81(18)
      abb81(40)=-abb81(38)*spbk2e1
      abb81(39)=abb81(39)+abb81(40)
      abb81(40)=spak2l4*mH**2*abb81(7)*abb81(6)
      abb81(39)=abb81(40)*abb81(20)*abb81(39)
      abb81(12)=-abb81(24)*abb81(12)
      abb81(24)=abb81(34)*abb81(12)
      abb81(41)=abb81(24)*spae1l3
      abb81(42)=abb81(41)*spae2k2
      abb81(15)=abb81(15)*spae1l4
      abb81(43)=abb81(15)*spae2k2
      abb81(44)=abb81(43)*spbe2e1
      abb81(42)=abb81(42)+abb81(44)
      abb81(44)=-spak1l5*spbk2k1*abb81(42)
      abb81(23)=-abb81(37)*abb81(23)**2
      abb81(30)=abb81(33)*abb81(23)*abb81(30)
      abb81(45)=spbk1e1*abb81(30)
      abb81(24)=abb81(24)*spae1l5
      abb81(46)=spae2k2*abb81(24)
      abb81(47)=spbk2k1*abb81(46)
      abb81(45)=abb81(47)+abb81(45)
      abb81(45)=spak1l3*abb81(45)
      abb81(47)=abb81(17)*spbk2e1
      abb81(38)=abb81(47)*abb81(38)
      abb81(11)=-spae2l4*spae1l5*spbe2e1*abb81(11)
      abb81(36)=abb81(12)*abb81(36)
      abb81(48)=abb81(36)*spbk2e1
      abb81(49)=abb81(48)*spbl3k1
      abb81(50)=-spak1k2*abb81(29)*abb81(49)
      abb81(36)=abb81(36)*spbl3e1
      abb81(51)=abb81(36)*abb81(29)
      abb81(52)=es12*abb81(51)
      abb81(8)=abb81(52)+abb81(50)+abb81(11)+abb81(45)+abb81(44)+abb81(39)+abb8&
      &1(8)+abb81(38)+abb81(32)+abb81(13)+abb81(28)
      abb81(11)=-2.0_ki*abb81(51)
      abb81(13)=abb81(19)*abb81(15)
      abb81(28)=spae2l5*abb81(41)
      abb81(32)=-spae2l4*abb81(26)
      abb81(35)=abb81(48)*abb81(35)
      abb81(38)=-spae2l3*abb81(24)
      abb81(13)=abb81(38)+abb81(35)+abb81(32)+abb81(28)+abb81(13)-abb81(51)
      abb81(28)=abb81(33)*spae1l3
      abb81(32)=abb81(28)*abb81(12)
      abb81(32)=abb81(32)+abb81(15)
      abb81(35)=spak1l5*abb81(32)
      abb81(38)=-spak1l4*abb81(25)
      abb81(39)=abb81(33)*spae1l5
      abb81(44)=abb81(39)*abb81(12)
      abb81(45)=-spak1l3*abb81(44)
      abb81(35)=abb81(45)+abb81(38)+abb81(35)
      abb81(35)=spbe2k1*abb81(35)
      abb81(14)=abb81(14)*spae1l4
      abb81(38)=-abb81(23)*abb81(28)
      abb81(38)=abb81(38)-abb81(14)
      abb81(38)=abb81(38)*spbk2e2*abb81(5)
      abb81(45)=-abb81(40)*abb81(18)
      abb81(50)=-spal3l4*abb81(22)
      abb81(45)=abb81(50)+abb81(45)
      abb81(50)=spak2l5*spbk2e2
      abb81(51)=abb81(50)*abb81(37)
      abb81(45)=abb81(51)*abb81(45)
      abb81(9)=abb81(9)*abb81(2)
      abb81(9)=abb81(9)+abb81(10)
      abb81(9)=-abb81(9)*abb81(16)
      abb81(10)=abb81(9)*abb81(4)
      abb81(16)=abb81(10)*abb81(50)
      abb81(50)=abb81(18)*abb81(16)
      abb81(35)=abb81(50)+abb81(45)+abb81(38)+abb81(35)
      abb81(38)=abb81(21)*spbl3e1
      abb81(20)=abb81(40)*abb81(20)
      abb81(45)=abb81(20)*spbk2e1
      abb81(38)=-abb81(47)+abb81(38)+abb81(45)
      abb81(45)=spae2l5*abb81(38)
      abb81(47)=abb81(36)*spbk2k1
      abb81(47)=abb81(47)-abb81(49)
      abb81(49)=-spak1e2*abb81(47)
      abb81(45)=abb81(49)+abb81(45)
      abb81(38)=spae1e2*abb81(38)
      abb81(43)=-spbk2e1*abb81(43)
      abb81(49)=spae2k2*spbk2e1
      abb81(12)=abb81(12)*abb81(49)
      abb81(28)=-abb81(28)*abb81(12)
      abb81(28)=abb81(28)+abb81(43)+abb81(38)
      abb81(38)=abb81(40)*abb81(37)
      abb81(10)=abb81(10)-abb81(38)
      abb81(18)=-abb81(18)*abb81(10)
      abb81(38)=abb81(37)*spal3l4
      abb81(22)=abb81(38)*abb81(22)
      abb81(18)=abb81(22)+abb81(18)-2.0_ki*abb81(32)
      abb81(22)=abb81(25)*abb81(49)
      abb81(25)=2.0_ki*abb81(25)
      abb81(32)=-2.0_ki*abb81(48)
      abb81(12)=abb81(39)*abb81(12)
      abb81(39)=2.0_ki*abb81(44)
      abb81(21)=-abb81(19)*abb81(21)
      abb81(43)=-spal3l4*abb81(51)
      abb81(14)=-spbe2e1*abb81(14)
      abb81(23)=-spae1l3*abb81(23)*abb81(34)
      abb81(14)=abb81(14)+abb81(23)
      abb81(14)=abb81(5)*abb81(14)
      abb81(23)=spae1k2*spbk2e2*abb81(36)
      abb81(14)=abb81(23)+abb81(14)
      abb81(23)=-spak1l4*abb81(9)
      abb81(33)=abb81(33)*abb81(37)*mT**2
      abb81(34)=spak1l3*abb81(33)
      abb81(23)=abb81(34)+abb81(23)
      abb81(23)=spbk1e1*abb81(5)*abb81(23)
      abb81(23)=2.0_ki*abb81(36)+abb81(23)
      abb81(9)=-abb81(5)*abb81(9)
      abb81(33)=abb81(5)*abb81(33)
      abb81(17)=abb81(17)-abb81(20)
      abb81(17)=abb81(19)*abb81(17)
      abb81(19)=-abb81(40)*abb81(51)
      abb81(16)=abb81(16)+abb81(19)
      abb81(15)=abb81(15)*spbe2e1
      abb81(15)=-abb81(15)-abb81(41)
      abb81(15)=spak1l5*abb81(15)
      abb81(19)=spak1l4*abb81(26)
      abb81(20)=spak1l3*abb81(24)
      abb81(15)=abb81(20)+abb81(19)+abb81(15)
      abb81(19)=spae1e2*abb81(47)
      abb81(20)=abb81(29)*abb81(48)
      R2d81=0.0_ki
      rat2 = rat2 + R2d81
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='81' value='", &
          & R2d81, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd81h0_qp
