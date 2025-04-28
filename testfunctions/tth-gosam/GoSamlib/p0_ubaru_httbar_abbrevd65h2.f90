module     p0_ubaru_httbar_abbrevd65h2
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh2
   implicit none
   private
   complex(ki), dimension(31), public :: abb65
   complex(ki), public :: R2d65
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_color, only: TR
      use p0_ubaru_httbar_globalsl1, only: epspow
      implicit none
      abb65(1)=1.0_ki/(-mT**2+es34)
      abb65(2)=sqrt(mT**2)
      abb65(3)=NC**(-1)
      abb65(4)=spbl4k2**(-1)
      abb65(5)=spak2l3**(-1)
      abb65(6)=spbl3k2**(-1)
      abb65(7)=spbl5k2**(-1)
      abb65(8)=abb65(3)*c1
      abb65(9)=2.0_ki*c2
      abb65(10)=abb65(8)-abb65(9)
      abb65(11)=abb65(10)*abb65(3)
      abb65(12)=c2*NC
      abb65(11)=abb65(12)+abb65(11)
      abb65(13)=spbk2k1*abb65(4)
      abb65(14)=-abb65(11)*abb65(13)*abb65(2)
      abb65(15)=TR**2*abb65(1)*gHT*e*gs**4
      abb65(16)=abb65(15)*i_
      abb65(16)=4.0_ki*abb65(16)
      abb65(17)=mT*spak2l5
      abb65(18)=abb65(14)*abb65(17)*abb65(16)
      abb65(19)=c1*abb65(3)*abb65(7)
      abb65(20)=abb65(9)*abb65(7)
      abb65(19)=abb65(19)-abb65(20)
      abb65(21)=abb65(2)**2
      abb65(22)=abb65(21)*abb65(4)
      abb65(23)=spak2l4*mH**2*abb65(5)*abb65(6)
      abb65(22)=abb65(22)-abb65(23)
      abb65(24)=abb65(3)*abb65(22)*abb65(19)
      abb65(25)=NC*c2*abb65(7)
      abb65(22)=abb65(22)*abb65(25)
      abb65(22)=abb65(22)+abb65(24)
      abb65(22)=spbk2k1*abb65(22)
      abb65(24)=abb65(8)*abb65(7)
      abb65(26)=abb65(24)-abb65(20)
      abb65(26)=abb65(26)*abb65(3)
      abb65(27)=abb65(12)*abb65(7)
      abb65(26)=abb65(26)+abb65(27)
      abb65(28)=mT**2
      abb65(29)=-abb65(28)*abb65(13)*abb65(26)
      abb65(13)=-abb65(13)*abb65(11)
      abb65(30)=-spak2l5*abb65(13)
      abb65(22)=abb65(29)+abb65(30)+abb65(22)
      abb65(22)=mT*abb65(22)
      abb65(29)=spal3l4*spbl3k1
      abb65(30)=abb65(23)*spbk2k1
      abb65(29)=abb65(29)+abb65(30)
      abb65(31)=abb65(2)*abb65(29)
      abb65(19)=abb65(3)*abb65(31)*abb65(19)
      abb65(14)=abb65(14)*spak2l5
      abb65(25)=abb65(31)*abb65(25)
      abb65(19)=abb65(22)-abb65(14)+abb65(25)+abb65(19)
      abb65(19)=mT*abb65(19)
      abb65(22)=spak2l5*abb65(30)*abb65(11)
      abb65(19)=abb65(22)+abb65(19)
      abb65(19)=abb65(19)*abb65(16)
      abb65(10)=-abb65(3)*abb65(29)*abb65(10)
      abb65(22)=-abb65(29)*abb65(12)
      abb65(10)=abb65(22)+abb65(10)
      abb65(10)=spak2l5*abb65(10)
      abb65(13)=abb65(13)*abb65(17)
      abb65(13)=abb65(14)+abb65(13)
      abb65(13)=mT*abb65(13)
      abb65(10)=abb65(10)+abb65(13)
      abb65(10)=abb65(10)*abb65(16)
      abb65(8)=abb65(8)*spbl5k1
      abb65(13)=spbl5k1*abb65(9)
      abb65(13)=abb65(13)-abb65(8)
      abb65(13)=abb65(3)*abb65(13)
      abb65(14)=-abb65(12)*spbl5k1
      abb65(13)=abb65(14)+abb65(13)
      abb65(14)=2.0_ki*i_
      abb65(14)=abb65(14)*abb65(15)
      abb65(13)=abb65(14)*spak2l5*spal3l4*abb65(13)
      abb65(15)=abb65(23)*spbl5k1
      abb65(16)=abb65(9)*abb65(15)
      abb65(22)=-abb65(23)*abb65(8)
      abb65(16)=abb65(16)+abb65(22)
      abb65(16)=abb65(3)*abb65(16)
      abb65(15)=-abb65(12)*abb65(15)
      abb65(15)=abb65(15)+abb65(16)
      abb65(15)=spak2l5*abb65(15)
      abb65(16)=abb65(4)*spbl5k1
      abb65(12)=-abb65(12)*abb65(16)
      abb65(9)=abb65(9)*abb65(16)
      abb65(8)=-abb65(4)*abb65(8)
      abb65(8)=abb65(9)+abb65(8)
      abb65(8)=abb65(3)*abb65(8)
      abb65(8)=abb65(12)+abb65(8)
      abb65(9)=abb65(28)*spak2l5
      abb65(8)=abb65(8)*abb65(9)
      abb65(8)=abb65(15)+abb65(8)
      abb65(8)=abb65(8)*abb65(14)
      abb65(12)=spal3l4*spbl5l3
      abb65(15)=abb65(23)*spbl5k2
      abb65(12)=abb65(12)+abb65(15)
      abb65(12)=spak2l5*abb65(12)*abb65(11)
      abb65(15)=abb65(9)*abb65(11)*abb65(4)*spbl5k2
      abb65(12)=abb65(12)+abb65(15)
      abb65(12)=abb65(12)*abb65(14)
      abb65(15)=abb65(20)*abb65(2)
      abb65(16)=spbl5k1*abb65(15)
      abb65(22)=abb65(24)*spbl5k1
      abb65(23)=-abb65(2)*abb65(22)
      abb65(16)=abb65(16)+abb65(23)
      abb65(16)=abb65(3)*abb65(16)
      abb65(23)=abb65(27)*spbl5k1
      abb65(25)=-abb65(2)*abb65(23)
      abb65(16)=abb65(25)+abb65(16)
      abb65(16)=spak2l5*abb65(16)
      abb65(20)=abb65(20)*spbl5k1
      abb65(20)=abb65(20)-abb65(22)
      abb65(20)=abb65(3)*abb65(20)
      abb65(20)=-abb65(23)+abb65(20)
      abb65(20)=abb65(20)*abb65(17)
      abb65(16)=abb65(16)+abb65(20)
      abb65(20)=abb65(14)*mT
      abb65(16)=abb65(16)*abb65(20)
      abb65(21)=spak2l5*abb65(21)*abb65(11)
      abb65(22)=abb65(17)*abb65(2)*abb65(11)
      abb65(21)=abb65(21)+abb65(22)
      abb65(21)=abb65(21)*abb65(14)
      abb65(22)=abb65(4)*spbl3k2
      abb65(9)=-abb65(14)*abb65(9)*abb65(26)*abb65(22)*spbl5k1
      abb65(17)=abb65(14)*abb65(17)*abb65(11)*abb65(22)*abb65(2)
      abb65(23)=spbk2k1*spal3l4
      abb65(25)=-abb65(28)*abb65(23)*abb65(26)
      abb65(23)=spak2l5*abb65(23)*abb65(11)
      abb65(23)=abb65(23)+abb65(25)
      abb65(23)=abb65(23)*abb65(14)
      abb65(25)=spal3l4*spbl3k2
      abb65(29)=-abb65(28)*abb65(25)*abb65(26)
      abb65(11)=spak2l5*abb65(25)*abb65(11)
      abb65(11)=abb65(11)+abb65(29)
      abb65(11)=abb65(11)*abb65(14)
      abb65(24)=-abb65(2)*abb65(24)
      abb65(15)=abb65(15)+abb65(24)
      abb65(15)=abb65(3)*abb65(15)
      abb65(24)=-mT*abb65(26)
      abb65(25)=-abb65(2)*abb65(27)
      abb65(15)=abb65(24)+abb65(25)+abb65(15)
      abb65(15)=abb65(15)*abb65(20)
      abb65(14)=-abb65(28)*abb65(14)*abb65(22)*abb65(26)
      R2d65=0.0_ki
      rat2 = rat2 + R2d65
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='65' value='", &
          & R2d65, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd65h2
