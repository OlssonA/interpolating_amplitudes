module     p2_gg_httbar_abbrevd1h8
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh8
   implicit none
   private
   complex(ki), dimension(36), public :: abb1
   complex(ki), public :: R2d1
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
      abb1(1)=1.0_ki/(-mT**2+es34)
      abb1(2)=sqrt(mT**2)
      abb1(3)=spak2l3**(-1)
      abb1(4)=spbl3k2**(-1)
      abb1(5)=spbl4k2**(-1)
      abb1(6)=spak2l5**(-1)
      abb1(7)=mT**2
      abb1(8)=i_*TR*e*gHT*abb1(1)*gs**4
      abb1(9)=abb1(7)*abb1(8)
      abb1(10)=c3*abb1(2)
      abb1(11)=abb1(10)*abb1(5)
      abb1(12)=spbe2e1*abb1(6)
      abb1(13)=abb1(9)*abb1(11)*abb1(12)
      abb1(14)=1.0_ki/2.0_ki*c1
      abb1(15)=abb1(14)*abb1(5)
      abb1(16)=c2*abb1(5)
      abb1(17)=abb1(15)-abb1(16)
      abb1(18)=abb1(8)*abb1(2)
      abb1(19)=NC*abb1(18)
      abb1(7)=abb1(19)*abb1(7)
      abb1(20)=abb1(7)*abb1(12)
      abb1(21)=-abb1(20)*abb1(17)
      abb1(21)=1.0_ki/2.0_ki*abb1(13)+abb1(21)
      abb1(21)=spae1l3*spae2k2*abb1(21)
      abb1(22)=abb1(5)*abb1(6)
      abb1(23)=spbe2e1*spae1e2
      abb1(24)=abb1(9)*abb1(23)*abb1(10)*abb1(22)
      abb1(25)=c1*abb1(5)
      abb1(26)=abb1(25)+abb1(16)
      abb1(27)=abb1(23)*abb1(6)
      abb1(26)=abb1(27)*abb1(7)*abb1(26)
      abb1(24)=2.0_ki*abb1(24)+abb1(26)
      abb1(24)=spak2l3*abb1(24)
      abb1(26)=abb1(16)*abb1(20)
      abb1(13)=-abb1(13)+abb1(26)
      abb1(20)=-abb1(25)*abb1(20)
      abb1(13)=1.0_ki/2.0_ki*abb1(13)+abb1(20)
      abb1(13)=spae2l3*spae1k2*abb1(13)
      abb1(13)=abb1(24)+abb1(21)+abb1(13)
      abb1(13)=spbl3k2*abb1(13)
      abb1(20)=abb1(19)*spae1e2
      abb1(21)=abb1(20)*abb1(14)
      abb1(24)=abb1(8)*spae1e2
      abb1(26)=abb1(10)*abb1(24)
      abb1(28)=abb1(20)*c2
      abb1(21)=-1.0_ki/2.0_ki*abb1(26)+abb1(21)-abb1(28)
      abb1(29)=-spbl3e1*spbl5e2*abb1(21)
      abb1(30)=c1+c2
      abb1(31)=abb1(30)*abb1(23)*abb1(19)
      abb1(32)=abb1(26)*spbe2e1
      abb1(31)=2.0_ki*abb1(32)+abb1(31)
      abb1(32)=-spbl5l3*abb1(31)
      abb1(26)=abb1(28)-abb1(26)
      abb1(20)=abb1(20)*c1
      abb1(20)=-abb1(20)+1.0_ki/2.0_ki*abb1(26)
      abb1(26)=spbl3e2*spbl5e1*abb1(20)
      abb1(26)=abb1(32)+abb1(29)+abb1(26)
      abb1(26)=spal3l4*abb1(26)
      abb1(28)=abb1(9)*NC
      abb1(29)=abb1(19)*mT
      abb1(28)=abb1(29)+abb1(28)
      abb1(28)=abb1(28)*abb1(2)
      abb1(29)=abb1(28)*abb1(16)
      abb1(32)=abb1(8)*mT
      abb1(33)=abb1(32)*abb1(2)
      abb1(33)=abb1(33)+abb1(9)
      abb1(11)=abb1(11)*abb1(33)
      abb1(34)=-abb1(28)*abb1(15)
      abb1(34)=abb1(34)+1.0_ki/2.0_ki*abb1(11)+abb1(29)
      abb1(34)=spae1e2*abb1(34)
      abb1(35)=abb1(4)*mH**2*abb1(3)*spak2l4
      abb1(21)=-abb1(21)*abb1(35)
      abb1(21)=abb1(34)+abb1(21)
      abb1(21)=spbk2e1*spbl5e2*abb1(21)
      abb1(34)=abb1(28)*abb1(25)
      abb1(36)=-abb1(34)-2.0_ki*abb1(11)-abb1(29)
      abb1(36)=abb1(23)*abb1(36)
      abb1(31)=-abb1(31)*abb1(35)
      abb1(31)=abb1(31)+abb1(36)
      abb1(31)=spbl5k2*abb1(31)
      abb1(36)=abb1(28)*abb1(30)
      abb1(10)=abb1(33)*abb1(10)
      abb1(33)=2.0_ki*abb1(10)+abb1(36)
      abb1(33)=spak2l4*abb1(27)*abb1(33)
      abb1(11)=-abb1(11)+abb1(29)
      abb1(11)=1.0_ki/2.0_ki*abb1(11)-abb1(34)
      abb1(11)=spae1e2*abb1(11)
      abb1(20)=abb1(20)*abb1(35)
      abb1(11)=abb1(20)+abb1(11)
      abb1(11)=spbk2e2*spbl5e1*abb1(11)
      abb1(10)=abb1(10)*abb1(12)
      abb1(20)=abb1(28)*abb1(12)
      abb1(28)=c2*abb1(20)
      abb1(28)=-abb1(10)+abb1(28)
      abb1(29)=-c1*abb1(20)
      abb1(28)=1.0_ki/2.0_ki*abb1(28)+abb1(29)
      abb1(28)=spae2l4*spae1k2*abb1(28)
      abb1(29)=-abb1(14)+c2
      abb1(20)=abb1(20)*abb1(29)
      abb1(10)=1.0_ki/2.0_ki*abb1(10)+abb1(20)
      abb1(10)=spae1l4*spae2k2*abb1(10)
      abb1(10)=abb1(31)+abb1(10)+abb1(28)+abb1(21)+abb1(33)+abb1(11)+abb1(13)+a&
      &bb1(26)
      abb1(11)=abb1(8)*mT**3
      abb1(13)=abb1(11)*NC
      abb1(7)=abb1(13)+abb1(7)
      abb1(13)=abb1(7)*abb1(25)
      abb1(9)=abb1(9)*abb1(2)
      abb1(9)=abb1(9)+abb1(11)
      abb1(11)=c3*abb1(5)
      abb1(9)=abb1(9)*abb1(11)
      abb1(20)=abb1(7)*abb1(16)
      abb1(21)=-abb1(13)-2.0_ki*abb1(9)-abb1(20)
      abb1(21)=abb1(27)*abb1(21)
      abb1(12)=abb1(12)*abb1(24)
      abb1(26)=mT*abb1(12)*c3
      abb1(27)=NC*mT
      abb1(12)=abb1(12)*abb1(27)
      abb1(28)=-abb1(12)*abb1(30)
      abb1(28)=-2.0_ki*abb1(26)+abb1(28)
      abb1(28)=abb1(28)*abb1(35)
      abb1(21)=abb1(28)+abb1(21)
      abb1(28)=-abb1(9)+abb1(20)
      abb1(13)=1.0_ki/2.0_ki*abb1(28)-abb1(13)
      abb1(13)=abb1(6)*abb1(13)
      abb1(8)=abb1(27)*abb1(8)
      abb1(28)=abb1(6)*abb1(8)
      abb1(29)=abb1(28)*c2
      abb1(30)=abb1(6)*abb1(32)*c3
      abb1(31)=abb1(30)-abb1(29)
      abb1(33)=abb1(28)*c1
      abb1(31)=abb1(33)+1.0_ki/2.0_ki*abb1(31)
      abb1(33)=-abb1(31)*abb1(35)
      abb1(13)=abb1(33)+abb1(13)
      abb1(13)=spbk2e2*abb1(13)
      abb1(31)=-spbl3e2*spal3l4*abb1(31)
      abb1(13)=abb1(31)+abb1(13)
      abb1(13)=spae1k2*abb1(13)
      abb1(19)=abb1(8)+abb1(19)
      abb1(31)=abb1(19)*abb1(14)
      abb1(18)=abb1(18)+abb1(32)
      abb1(18)=abb1(18)*c3
      abb1(33)=abb1(19)*c2
      abb1(34)=-abb1(31)+1.0_ki/2.0_ki*abb1(18)+abb1(33)
      abb1(34)=spae1l4*abb1(34)
      abb1(17)=-abb1(8)*abb1(17)
      abb1(32)=abb1(11)*abb1(32)
      abb1(17)=1.0_ki/2.0_ki*abb1(32)+abb1(17)
      abb1(17)=spae1l3*spbl3k2*abb1(17)
      abb1(17)=abb1(17)+abb1(34)
      abb1(17)=spbl5e2*abb1(17)
      abb1(13)=abb1(17)+abb1(13)
      abb1(7)=abb1(7)*abb1(14)*abb1(22)
      abb1(17)=abb1(28)*abb1(14)
      abb1(17)=abb1(29)-abb1(17)+1.0_ki/2.0_ki*abb1(30)
      abb1(22)=-abb1(17)*abb1(35)
      abb1(28)=1.0_ki/2.0_ki*abb1(6)
      abb1(9)=-abb1(28)*abb1(9)
      abb1(20)=-abb1(6)*abb1(20)
      abb1(7)=abb1(22)+abb1(7)+abb1(9)+abb1(20)
      abb1(7)=spbk2e1*abb1(7)
      abb1(9)=-spbl3e1*spal3l4*abb1(17)
      abb1(7)=abb1(9)+abb1(7)
      abb1(7)=spae2k2*abb1(7)
      abb1(9)=abb1(18)-abb1(33)
      abb1(17)=c1*abb1(19)
      abb1(9)=1.0_ki/2.0_ki*abb1(9)+abb1(17)
      abb1(9)=spae2l4*abb1(9)
      abb1(17)=-abb1(8)*abb1(16)
      abb1(17)=abb1(32)+abb1(17)
      abb1(8)=abb1(8)*abb1(25)
      abb1(8)=1.0_ki/2.0_ki*abb1(17)+abb1(8)
      abb1(8)=spae2l3*spbl3k2*abb1(8)
      abb1(8)=abb1(8)+abb1(9)
      abb1(8)=spbl5e1*abb1(8)
      abb1(7)=abb1(8)+abb1(7)
      abb1(8)=abb1(31)+abb1(18)+1.0_ki/2.0_ki*abb1(33)
      abb1(8)=abb1(23)*abb1(8)
      abb1(9)=abb1(15)+1.0_ki/2.0_ki*abb1(16)
      abb1(9)=abb1(9)*abb1(27)
      abb1(11)=mT*abb1(11)
      abb1(9)=abb1(11)+abb1(9)
      abb1(9)=spbl3k2*spbe2e1*abb1(24)*abb1(9)
      abb1(11)=-abb1(14)-1.0_ki/2.0_ki*c2
      abb1(11)=abb1(12)*abb1(11)
      abb1(11)=-abb1(26)+abb1(11)
      abb1(11)=spal3l4*abb1(11)
      R2d1=0.0_ki
      rat2 = rat2 + R2d1
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='1' value='", &
          & R2d1, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd1h8
