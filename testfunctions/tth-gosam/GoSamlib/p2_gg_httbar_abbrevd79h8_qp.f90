module     p2_gg_httbar_abbrevd79h8_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh8_qp
   implicit none
   private
   complex(ki), dimension(52), public :: abb79
   complex(ki), public :: R2d79
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
      abb79(1)=1.0_ki/(es34-es51-es12)
      abb79(2)=sqrt(mT**2)
      abb79(3)=NC**(-1)
      abb79(4)=spak2l3**(-1)
      abb79(5)=spbl3k2**(-1)
      abb79(6)=spak2l5**(-1)
      abb79(7)=spbl5k2**(-1)
      abb79(8)=spbl4k2**(-1)
      abb79(9)=spbl5e2*abb79(1)
      abb79(10)=abb79(9)*mT
      abb79(11)=c1*e*gHT*abb79(3)*gs**4*i_*TR
      abb79(12)=abb79(10)*abb79(11)
      abb79(13)=abb79(8)*abb79(12)
      abb79(14)=abb79(2)**2
      abb79(15)=abb79(13)*abb79(14)
      abb79(16)=spae1k1*spbk2k1
      abb79(17)=abb79(16)*abb79(15)
      abb79(18)=abb79(11)*abb79(9)
      abb79(19)=abb79(2)**3
      abb79(20)=abb79(19)*abb79(18)
      abb79(21)=abb79(20)*spae1l4
      abb79(22)=abb79(21)-abb79(17)
      abb79(22)=spae2l3*abb79(22)
      abb79(11)=abb79(11)*abb79(6)*spae2k2
      abb79(23)=abb79(11)*abb79(14)*abb79(10)
      abb79(24)=abb79(23)*spae1l4
      abb79(25)=abb79(11)*mT**2
      abb79(26)=abb79(25)*abb79(8)
      abb79(27)=abb79(9)*abb79(2)
      abb79(28)=abb79(26)*abb79(27)
      abb79(29)=abb79(28)*abb79(16)
      abb79(30)=-abb79(24)+abb79(29)
      abb79(30)=spal3l5*abb79(30)
      abb79(31)=abb79(23)*spae1l5
      abb79(32)=abb79(20)*spae1e2
      abb79(31)=abb79(32)+abb79(31)
      abb79(32)=spal3l4*abb79(31)
      abb79(22)=abb79(32)+abb79(30)+abb79(22)
      abb79(22)=spbl3e1*abb79(22)
      abb79(30)=abb79(26)*abb79(1)
      abb79(32)=abb79(30)*abb79(19)
      abb79(33)=spbk2e1*abb79(32)
      abb79(34)=mT*abb79(11)*abb79(1)
      abb79(35)=abb79(34)*abb79(14)
      abb79(36)=spak1l4*spbk1e1
      abb79(37)=-abb79(36)*abb79(35)
      abb79(37)=-abb79(33)+abb79(37)
      abb79(37)=spbl3e2*abb79(37)
      abb79(38)=abb79(13)*spae2l5
      abb79(39)=abb79(38)*spbk2e1
      abb79(40)=abb79(39)*abb79(14)
      abb79(41)=abb79(18)*abb79(2)
      abb79(42)=abb79(41)*spae2l5
      abb79(43)=abb79(42)*abb79(36)
      abb79(44)=-abb79(40)-abb79(43)
      abb79(44)=spbl5l3*abb79(44)
      abb79(37)=abb79(44)+abb79(37)
      abb79(37)=spae1l3*abb79(37)
      abb79(44)=-abb79(14)*abb79(12)
      abb79(44)=abb79(20)+abb79(44)
      abb79(44)=spae1l4*abb79(44)
      abb79(18)=-abb79(14)*abb79(18)
      abb79(12)=abb79(2)*abb79(12)
      abb79(12)=abb79(18)+abb79(12)
      abb79(12)=mT*abb79(12)*abb79(16)*abb79(8)
      abb79(12)=abb79(12)+abb79(44)
      abb79(18)=mH**2*abb79(5)*abb79(4)
      abb79(44)=abb79(18)*spbk2e1
      abb79(12)=abb79(44)*abb79(12)
      abb79(45)=spbk2e1*abb79(21)
      abb79(12)=abb79(45)+abb79(12)
      abb79(12)=spae2k2*abb79(12)
      abb79(45)=-spak1l5*abb79(24)
      abb79(46)=-spak1e2*abb79(21)
      abb79(45)=abb79(45)+abb79(46)
      abb79(45)=spbk1e1*abb79(45)
      abb79(46)=abb79(2)**4
      abb79(13)=abb79(46)*abb79(13)*spbk2e1
      abb79(20)=abb79(36)*abb79(20)
      abb79(13)=abb79(13)+abb79(20)
      abb79(13)=spae1e2*abb79(13)
      abb79(20)=abb79(36)*abb79(23)
      abb79(9)=abb79(19)*abb79(9)
      abb79(19)=abb79(26)*spbk2e1
      abb79(23)=abb79(19)*abb79(9)
      abb79(20)=abb79(23)+abb79(20)
      abb79(20)=spae1l5*abb79(20)
      abb79(23)=spae1l4*abb79(46)*abb79(34)
      abb79(26)=-abb79(16)*abb79(32)
      abb79(23)=abb79(23)+abb79(26)
      abb79(23)=spbe2e1*abb79(23)
      abb79(26)=-spae2l5*abb79(21)
      abb79(34)=abb79(16)*abb79(38)
      abb79(46)=abb79(14)*abb79(34)
      abb79(26)=abb79(26)+abb79(46)
      abb79(26)=spbl5e1*abb79(26)
      abb79(32)=spbe2e1*abb79(32)
      abb79(38)=abb79(38)*spbl5e1
      abb79(46)=abb79(38)*abb79(14)
      abb79(32)=abb79(32)-abb79(46)
      abb79(46)=abb79(41)*spae2k2
      abb79(47)=-abb79(36)*abb79(46)
      abb79(47)=abb79(47)+abb79(32)
      abb79(48)=spbl3k2*spae1l3
      abb79(47)=abb79(47)*abb79(48)
      abb79(49)=abb79(25)*abb79(27)
      abb79(48)=abb79(36)*abb79(49)*abb79(48)
      abb79(25)=spae1l4*abb79(25)*spbk2e1
      abb79(9)=-abb79(25)*abb79(9)
      abb79(9)=abb79(9)+abb79(48)
      abb79(9)=abb79(7)*abb79(9)
      abb79(44)=spak2l4*abb79(44)*abb79(31)
      abb79(48)=spbe2k1*spae1k1*abb79(33)
      abb79(50)=abb79(39)*spbl5k1
      abb79(51)=abb79(50)*spae1k1
      abb79(14)=-abb79(14)*abb79(51)
      abb79(52)=spbl5k2*abb79(18)*spae1k2
      abb79(43)=-abb79(43)*abb79(52)
      abb79(9)=abb79(44)+abb79(43)+abb79(14)+abb79(48)+abb79(9)+abb79(47)+abb79&
      &(26)+abb79(23)+abb79(20)+abb79(13)+abb79(22)+abb79(12)+abb79(37)+abb79(4&
      &5)
      abb79(12)=abb79(41)*spae1l4
      abb79(13)=abb79(12)*spak1e2
      abb79(10)=abb79(11)*abb79(10)
      abb79(11)=abb79(10)*spae1l4
      abb79(14)=abb79(11)*spak1l5
      abb79(13)=abb79(13)+abb79(14)
      abb79(14)=-spbk1e1*abb79(13)
      abb79(20)=abb79(15)*spbk2e1
      abb79(22)=abb79(41)*abb79(36)
      abb79(20)=abb79(20)+abb79(22)
      abb79(20)=spae1e2*abb79(20)
      abb79(19)=abb79(19)*abb79(27)
      abb79(22)=abb79(10)*abb79(36)
      abb79(22)=abb79(19)+abb79(22)
      abb79(22)=spae1l5*abb79(22)
      abb79(23)=spae1l4*abb79(35)
      abb79(26)=abb79(30)*abb79(2)
      abb79(16)=-abb79(16)*abb79(26)
      abb79(16)=abb79(23)+abb79(16)
      abb79(16)=spbe2e1*abb79(16)
      abb79(23)=abb79(12)*spae2l5
      abb79(30)=-abb79(23)+abb79(34)
      abb79(30)=spbl5e1*abb79(30)
      abb79(34)=abb79(12)*spae2k2
      abb79(37)=spbk2e1*abb79(34)
      abb79(25)=-abb79(7)*abb79(27)*abb79(25)
      abb79(27)=abb79(26)*spbk2e1
      abb79(43)=abb79(27)*spbe2k1
      abb79(44)=spae1k1*abb79(43)
      abb79(14)=-abb79(51)+abb79(44)+abb79(25)+abb79(30)+abb79(16)+abb79(22)+ab&
      &b79(37)+abb79(20)+abb79(14)
      abb79(16)=3.0_ki*abb79(21)-2.0_ki*abb79(17)
      abb79(17)=2.0_ki*abb79(36)
      abb79(20)=-abb79(35)*abb79(17)
      abb79(20)=-3.0_ki*abb79(33)+abb79(20)
      abb79(21)=abb79(42)*abb79(17)
      abb79(21)=3.0_ki*abb79(40)+abb79(21)
      abb79(22)=3.0_ki*abb79(24)-2.0_ki*abb79(29)
      abb79(24)=abb79(35)*spbl3e2
      abb79(25)=abb79(49)*abb79(7)
      abb79(29)=abb79(25)*spbl3k2
      abb79(24)=abb79(24)-abb79(29)
      abb79(29)=spbl3k2*abb79(46)
      abb79(30)=spbl5l3*abb79(42)
      abb79(29)=abb79(30)+abb79(29)+abb79(24)
      abb79(29)=spae1l3*abb79(29)
      abb79(30)=abb79(42)*abb79(52)
      abb79(29)=abb79(30)-3.0_ki*abb79(31)+abb79(29)
      abb79(30)=abb79(41)*spae1e2
      abb79(10)=abb79(10)*spae1l5
      abb79(10)=abb79(30)+abb79(10)
      abb79(30)=2.0_ki*abb79(35)
      abb79(31)=-2.0_ki*abb79(42)
      abb79(33)=abb79(43)-abb79(50)
      abb79(26)=abb79(26)*spbe2e1
      abb79(26)=abb79(26)-abb79(38)
      abb79(35)=spbk2k1*abb79(26)
      abb79(35)=abb79(35)-abb79(33)
      abb79(35)=spak1l3*abb79(35)
      abb79(36)=abb79(15)*spae2l3
      abb79(37)=-spbk2e1*abb79(36)
      abb79(19)=spal3l5*abb79(19)
      abb79(19)=abb79(35)+abb79(37)+abb79(19)
      abb79(35)=-spak1l4*abb79(10)
      abb79(13)=abb79(13)+abb79(35)
      abb79(35)=spbl3k1*abb79(13)
      abb79(24)=-spae1l4*abb79(24)
      abb79(34)=-spbl3k2*abb79(34)
      abb79(37)=-spbl5l3*abb79(23)
      abb79(24)=abb79(35)+abb79(37)+abb79(34)+abb79(24)
      abb79(34)=-spal3l5*abb79(28)
      abb79(34)=abb79(34)+abb79(36)
      abb79(34)=spbl3e1*abb79(34)
      abb79(25)=abb79(46)-abb79(25)
      abb79(17)=-abb79(17)*abb79(25)
      abb79(33)=-spak1k2*abb79(18)*abb79(33)
      abb79(35)=-abb79(18)*abb79(26)
      abb79(36)=-es12*abb79(35)
      abb79(17)=abb79(36)+abb79(33)+abb79(17)+abb79(34)+3.0_ki*abb79(32)
      abb79(32)=-2.0_ki*abb79(35)
      abb79(15)=2.0_ki*abb79(15)
      abb79(28)=2.0_ki*abb79(28)
      abb79(25)=2.0_ki*abb79(25)
      abb79(33)=-abb79(18)*abb79(27)
      abb79(13)=spbk2k1*abb79(13)
      abb79(23)=-spbl5k2*abb79(23)
      abb79(13)=abb79(23)+abb79(13)
      abb79(13)=abb79(18)*abb79(13)
      abb79(23)=abb79(18)*abb79(12)
      abb79(34)=abb79(18)*abb79(11)
      abb79(35)=-abb79(18)*abb79(10)
      abb79(18)=abb79(18)*abb79(39)
      R2d79=0.0_ki
      rat2 = rat2 + R2d79
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='79' value='", &
          & R2d79, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd79h8_qp
