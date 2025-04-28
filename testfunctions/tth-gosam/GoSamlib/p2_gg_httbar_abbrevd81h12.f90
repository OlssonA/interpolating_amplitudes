module     p2_gg_httbar_abbrevd81h12
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh12
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
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb81(1)=1.0_ki/(-mT**2+es34)
      abb81(2)=sqrt(mT**2)
      abb81(3)=NC**(-1)
      abb81(4)=spak2l5**(-1)
      abb81(5)=spak2l4**(-1)
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
      abb81(13)=-spbl4e1*abb81(11)
      abb81(14)=abb81(8)*abb81(9)
      abb81(15)=abb81(9)*abb81(2)**2
      abb81(16)=abb81(15)*mT
      abb81(14)=abb81(16)+abb81(14)
      abb81(16)=-mT*abb81(12)
      abb81(14)=-abb81(14)*abb81(16)
      abb81(17)=abb81(14)*abb81(5)
      abb81(18)=spbk1e1*spak1k2
      abb81(19)=-abb81(18)*abb81(17)
      abb81(20)=-abb81(15)*abb81(12)
      abb81(21)=abb81(20)*spbl4l3
      abb81(22)=spak1l3*spbk1e1
      abb81(23)=abb81(21)*abb81(22)
      abb81(13)=abb81(23)+abb81(13)+abb81(19)
      abb81(19)=spbl5e2*spae1e2
      abb81(13)=abb81(19)*abb81(13)
      abb81(23)=mT*abb81(2)
      abb81(24)=abb81(23)*abb81(9)
      abb81(15)=abb81(15)+abb81(24)
      abb81(15)=-abb81(15)*abb81(12)
      abb81(25)=abb81(15)*spbl5e1
      abb81(26)=abb81(25)*spae1e2
      abb81(27)=abb81(26)*spbk2e2
      abb81(28)=-spak1k2*abb81(27)
      abb81(29)=spbe2e1*spae2k2
      abb81(30)=abb81(4)*abb81(29)
      abb81(31)=abb81(30)*abb81(14)
      abb81(32)=-spae1k1*abb81(31)
      abb81(28)=abb81(32)+abb81(28)
      abb81(28)=spbl4k1*abb81(28)
      abb81(32)=spbl3e1*spbl5e2
      abb81(33)=-spbl3e2*spbl5e1
      abb81(32)=abb81(33)+abb81(32)
      abb81(33)=spak2l3*abb81(5)
      abb81(34)=abb81(33)*spae1e2
      abb81(32)=abb81(34)*abb81(32)
      abb81(35)=spae2l3*spbe2e1
      abb81(36)=spbl4l3*abb81(4)
      abb81(37)=spae1k2*abb81(36)*abb81(35)
      abb81(32)=abb81(37)+abb81(32)
      abb81(37)=-abb81(9)*abb81(12)
      abb81(8)=-abb81(37)*abb81(8)*mT
      abb81(32)=abb81(8)*abb81(32)
      abb81(8)=-abb81(8)*abb81(36)*abb81(29)
      abb81(38)=abb81(29)*spbl5k2
      abb81(39)=abb81(21)*abb81(38)
      abb81(8)=abb81(8)+abb81(39)
      abb81(8)=spae1l3*abb81(8)
      abb81(39)=abb81(19)*abb81(18)
      abb81(40)=abb81(38)*spae1k2
      abb81(39)=abb81(39)+abb81(40)
      abb81(40)=spbl4k2*mH**2*abb81(7)*abb81(6)
      abb81(39)=abb81(40)*abb81(20)*abb81(39)
      abb81(12)=-abb81(24)*abb81(12)
      abb81(24)=abb81(34)*abb81(12)
      abb81(41)=abb81(24)*spbl3e1
      abb81(42)=abb81(41)*spbk2e2
      abb81(15)=abb81(15)*spbl4e1
      abb81(43)=abb81(15)*spbk2e2
      abb81(44)=abb81(43)*spae1e2
      abb81(42)=abb81(42)+abb81(44)
      abb81(44)=spbl5k1*spak1k2*abb81(42)
      abb81(23)=-abb81(37)*abb81(23)**2
      abb81(30)=abb81(33)*abb81(23)*abb81(30)
      abb81(45)=-spae1k1*abb81(30)
      abb81(24)=abb81(24)*spbl5e1
      abb81(46)=spbk2e2*abb81(24)
      abb81(47)=-spak1k2*abb81(46)
      abb81(45)=abb81(47)+abb81(45)
      abb81(45)=spbl3k1*abb81(45)
      abb81(47)=abb81(17)*spae1k2
      abb81(38)=-abb81(47)*abb81(38)
      abb81(11)=spbl4e2*spbl5e1*spae1e2*abb81(11)
      abb81(36)=abb81(12)*abb81(36)
      abb81(48)=abb81(36)*spae1k2
      abb81(49)=abb81(48)*spak1l3
      abb81(50)=spbk2k1*abb81(29)*abb81(49)
      abb81(36)=abb81(36)*spae1l3
      abb81(51)=abb81(36)*abb81(29)
      abb81(52)=-es12*abb81(51)
      abb81(8)=abb81(52)+abb81(50)+abb81(11)+abb81(45)+abb81(44)+abb81(39)+abb8&
      &1(8)+abb81(38)+abb81(32)+abb81(13)+abb81(28)
      abb81(11)=2.0_ki*abb81(51)
      abb81(13)=-abb81(19)*abb81(15)
      abb81(28)=-spbl5e2*abb81(41)
      abb81(32)=spbl4e2*abb81(26)
      abb81(38)=spbl3e2*abb81(24)
      abb81(35)=-abb81(48)*abb81(35)
      abb81(13)=abb81(35)+abb81(38)+abb81(32)+abb81(28)+abb81(13)+abb81(51)
      abb81(28)=abb81(21)*spae1l3
      abb81(20)=abb81(40)*abb81(20)
      abb81(32)=abb81(20)*spae1k2
      abb81(28)=-abb81(47)+abb81(28)+abb81(32)
      abb81(32)=-spbl5e2*abb81(28)
      abb81(35)=abb81(36)*spak1k2
      abb81(35)=abb81(35)-abb81(49)
      abb81(38)=spbe2k1*abb81(35)
      abb81(32)=abb81(38)+abb81(32)
      abb81(38)=abb81(33)*spbl3e1
      abb81(39)=abb81(38)*abb81(12)
      abb81(39)=abb81(39)+abb81(15)
      abb81(44)=-spbl5k1*abb81(39)
      abb81(45)=spbl4k1*abb81(25)
      abb81(47)=abb81(33)*spbl5e1
      abb81(49)=abb81(47)*abb81(12)
      abb81(50)=spbl3k1*abb81(49)
      abb81(44)=abb81(50)+abb81(45)+abb81(44)
      abb81(44)=spak1e2*abb81(44)
      abb81(14)=abb81(14)*spbl4e1
      abb81(45)=abb81(23)*abb81(38)
      abb81(45)=abb81(45)+abb81(14)
      abb81(45)=abb81(45)*spae2k2*abb81(4)
      abb81(50)=abb81(40)*abb81(18)
      abb81(51)=spbl4l3*abb81(22)
      abb81(50)=abb81(51)+abb81(50)
      abb81(51)=spbl5k2*spae2k2
      abb81(52)=abb81(51)*abb81(37)
      abb81(50)=abb81(52)*abb81(50)
      abb81(9)=abb81(9)*abb81(2)
      abb81(9)=abb81(9)+abb81(10)
      abb81(9)=-abb81(9)*abb81(16)
      abb81(10)=abb81(9)*abb81(5)
      abb81(16)=abb81(10)*abb81(51)
      abb81(51)=-abb81(18)*abb81(16)
      abb81(44)=abb81(51)+abb81(50)+abb81(45)+abb81(44)
      abb81(28)=-spbe2e1*abb81(28)
      abb81(43)=spae1k2*abb81(43)
      abb81(45)=spbk2e2*spae1k2
      abb81(12)=abb81(12)*abb81(45)
      abb81(38)=abb81(38)*abb81(12)
      abb81(28)=abb81(38)+abb81(43)+abb81(28)
      abb81(38)=abb81(40)*abb81(37)
      abb81(10)=abb81(10)-abb81(38)
      abb81(18)=abb81(18)*abb81(10)
      abb81(38)=abb81(37)*spbl4l3
      abb81(22)=-abb81(38)*abb81(22)
      abb81(18)=abb81(22)+abb81(18)+2.0_ki*abb81(39)
      abb81(22)=-abb81(25)*abb81(45)
      abb81(25)=-2.0_ki*abb81(25)
      abb81(12)=-abb81(47)*abb81(12)
      abb81(39)=-2.0_ki*abb81(49)
      abb81(43)=2.0_ki*abb81(48)
      abb81(21)=abb81(19)*abb81(21)
      abb81(45)=spbl4l3*abb81(52)
      abb81(14)=spae1e2*abb81(14)
      abb81(23)=spbl3e1*abb81(23)*abb81(34)
      abb81(14)=abb81(14)+abb81(23)
      abb81(14)=abb81(4)*abb81(14)
      abb81(23)=-spbk2e1*spae2k2*abb81(36)
      abb81(14)=abb81(23)+abb81(14)
      abb81(23)=spbl4k1*abb81(9)
      abb81(33)=abb81(33)*abb81(37)*mT**2
      abb81(34)=-spbl3k1*abb81(33)
      abb81(23)=abb81(34)+abb81(23)
      abb81(23)=spae1k1*abb81(4)*abb81(23)
      abb81(23)=-2.0_ki*abb81(36)+abb81(23)
      abb81(9)=abb81(4)*abb81(9)
      abb81(33)=-abb81(4)*abb81(33)
      abb81(17)=-abb81(17)+abb81(20)
      abb81(17)=abb81(19)*abb81(17)
      abb81(19)=abb81(40)*abb81(52)
      abb81(16)=-abb81(16)+abb81(19)
      abb81(19)=-spbe2e1*abb81(35)
      abb81(15)=abb81(15)*spae1e2
      abb81(15)=abb81(15)+abb81(41)
      abb81(15)=spbl5k1*abb81(15)
      abb81(20)=-spbl4k1*abb81(26)
      abb81(24)=-spbl3k1*abb81(24)
      abb81(15)=abb81(24)+abb81(20)+abb81(15)
      abb81(20)=-abb81(29)*abb81(48)
      R2d81=0.0_ki
      rat2 = rat2 + R2d81
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='81' value='", &
          & R2d81, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd81h12
