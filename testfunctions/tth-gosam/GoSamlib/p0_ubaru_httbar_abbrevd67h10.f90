module     p0_ubaru_httbar_abbrevd67h10
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh10
   implicit none
   private
   complex(ki), dimension(31), public :: abb67
   complex(ki), public :: R2d67
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
      abb67(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb67(2)=NC**(-1)
      abb67(3)=spak2l5**(-1)
      abb67(4)=sqrt(mT**2)
      abb67(5)=spbl4k2**(-1)
      abb67(6)=spak2l3**(-1)
      abb67(7)=spbl3k2**(-1)
      abb67(8)=spak2l3*spbl3k1
      abb67(9)=abb67(3)*abb67(8)
      abb67(10)=abb67(9)+spbl5k1
      abb67(11)=abb67(2)*c1
      abb67(12)=abb67(10)*abb67(11)
      abb67(13)=c2*abb67(10)
      abb67(14)=2.0_ki*abb67(13)
      abb67(12)=abb67(12)-abb67(14)
      abb67(12)=abb67(12)*abb67(2)
      abb67(15)=abb67(10)*c1
      abb67(12)=abb67(12)+abb67(15)
      abb67(15)=mT*spak2l4
      abb67(12)=abb67(12)*abb67(15)
      abb67(16)=2.0_ki*c2
      abb67(17)=abb67(16)-abb67(11)
      abb67(18)=-abb67(4)*abb67(17)
      abb67(19)=abb67(18)*abb67(2)
      abb67(20)=c1*abb67(4)
      abb67(20)=abb67(19)+abb67(20)
      abb67(21)=-spak2l4*spbl5k1*abb67(20)
      abb67(12)=abb67(12)-abb67(21)
      abb67(22)=gs**4*TR**2*abb67(1)*gHT*e*i_
      abb67(23)=4.0_ki*abb67(22)
      abb67(12)=abb67(12)*abb67(23)
      abb67(24)=c1*abb67(5)
      abb67(25)=-abb67(10)*abb67(24)
      abb67(26)=abb67(11)*abb67(5)
      abb67(27)=-abb67(10)*abb67(26)
      abb67(14)=abb67(5)*abb67(14)
      abb67(14)=abb67(14)+abb67(27)
      abb67(14)=abb67(2)*abb67(14)
      abb67(17)=abb67(17)*abb67(2)
      abb67(27)=abb67(5)*abb67(3)
      abb67(28)=abb67(27)*spbl4k1
      abb67(29)=-abb67(28)*abb67(17)
      abb67(30)=abb67(24)*abb67(3)
      abb67(31)=spbl4k1*abb67(30)
      abb67(29)=abb67(31)+abb67(29)
      abb67(29)=spak2l4*abb67(29)
      abb67(14)=abb67(29)+abb67(25)+abb67(14)
      abb67(14)=mT*abb67(14)
      abb67(9)=abb67(4)*abb67(24)*abb67(9)
      abb67(18)=abb67(2)*abb67(18)*abb67(27)
      abb67(8)=abb67(18)*abb67(8)
      abb67(19)=abb67(28)*abb67(19)
      abb67(25)=abb67(30)*abb67(4)
      abb67(28)=spbl4k1*abb67(25)
      abb67(19)=abb67(28)+abb67(19)
      abb67(19)=spak2l4*abb67(19)
      abb67(8)=abb67(14)+abb67(19)+abb67(9)+abb67(8)
      abb67(8)=mT*abb67(8)
      abb67(9)=spbl5k2*mH**2*abb67(6)*abb67(7)
      abb67(14)=abb67(5)*spbl4k1
      abb67(19)=abb67(9)*abb67(14)
      abb67(28)=-c2*abb67(19)
      abb67(13)=abb67(28)-abb67(13)
      abb67(10)=abb67(19)+abb67(10)
      abb67(11)=abb67(10)*abb67(11)
      abb67(11)=2.0_ki*abb67(13)+abb67(11)
      abb67(11)=abb67(2)*abb67(11)
      abb67(10)=c1*abb67(10)
      abb67(10)=abb67(10)+abb67(11)
      abb67(10)=spak2l4*abb67(10)
      abb67(11)=abb67(16)*abb67(5)
      abb67(11)=abb67(11)-abb67(26)
      abb67(11)=abb67(11)*abb67(2)
      abb67(13)=abb67(11)-abb67(24)
      abb67(16)=abb67(4)**2
      abb67(19)=-abb67(13)*abb67(16)*spbl5k1
      abb67(8)=abb67(8)+abb67(10)+abb67(19)
      abb67(8)=mT*abb67(8)
      abb67(8)=-abb67(21)+abb67(8)
      abb67(8)=abb67(8)*abb67(23)
      abb67(10)=-abb67(14)*spbl5l3*abb67(17)
      abb67(14)=abb67(24)*spbl5l3
      abb67(19)=spbl4k1*abb67(14)
      abb67(10)=abb67(19)+abb67(10)
      abb67(19)=2.0_ki*abb67(22)
      abb67(10)=abb67(10)*abb67(15)*abb67(19)
      abb67(21)=-spak2l4*abb67(19)*spbl5l3*abb67(20)
      abb67(22)=-c1+abb67(17)
      abb67(16)=spak2l4*abb67(22)*abb67(16)*abb67(3)
      abb67(15)=-abb67(15)*abb67(3)*abb67(20)
      abb67(15)=abb67(16)+abb67(15)
      abb67(15)=mT*abb67(15)
      abb67(16)=-spak2l4*abb67(9)*abb67(20)
      abb67(15)=abb67(16)+abb67(15)
      abb67(15)=abb67(15)*abb67(19)
      abb67(16)=-abb67(27)*abb67(17)
      abb67(16)=abb67(30)+abb67(16)
      abb67(16)=mT*abb67(16)
      abb67(16)=abb67(16)+abb67(25)+abb67(18)
      abb67(16)=mT*abb67(16)
      abb67(9)=-abb67(9)*abb67(13)
      abb67(9)=abb67(16)+abb67(9)
      abb67(9)=mT*abb67(9)*abb67(23)
      abb67(11)=-spbl5l3*abb67(11)
      abb67(11)=abb67(14)+abb67(11)
      abb67(11)=mT*abb67(11)*abb67(19)
      R2d67=0.0_ki
      rat2 = rat2 + R2d67
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='67' value='", &
          & R2d67, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd67h10
