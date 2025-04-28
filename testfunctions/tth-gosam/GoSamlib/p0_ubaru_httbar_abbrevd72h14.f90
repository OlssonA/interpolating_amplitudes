module     p0_ubaru_httbar_abbrevd72h14
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh14
   implicit none
   private
   complex(ki), dimension(16), public :: abb72
   complex(ki), public :: R2d72
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
      abb72(1)=1.0_ki/(-mT**2+es34)
      abb72(2)=NC**(-1)
      abb72(3)=es12**(-1)
      abb72(4)=sqrt(mT**2)
      abb72(5)=spbl3k2**(-1)
      abb72(6)=spak2l4**(-1)
      abb72(7)=abb72(2)**2
      abb72(7)=abb72(7)-1.0_ki
      abb72(8)=TR**2*gs**4*i_*spbl5k1*e*gHT*abb72(3)*abb72(1)
      abb72(7)=abb72(7)*abb72(8)
      abb72(9)=c1*spak2l3*abb72(7)
      abb72(10)=spbl4l3*abb72(9)
      abb72(8)=abb72(8)*c2
      abb72(11)=spak2l3*abb72(8)
      abb72(12)=abb72(11)*spbl4l3
      abb72(13)=abb72(12)*NC
      abb72(12)=abb72(12)*abb72(2)
      abb72(10)=abb72(13)+abb72(10)-abb72(12)
      abb72(12)=-2.0_ki*abb72(10)
      abb72(13)=mT*abb72(4)
      abb72(14)=abb72(4)**2
      abb72(15)=abb72(13)+abb72(14)
      abb72(15)=2.0_ki*abb72(15)*abb72(10)
      abb72(10)=4.0_ki*abb72(10)
      abb72(7)=c1*abb72(7)
      abb72(16)=abb72(8)*NC
      abb72(8)=abb72(2)*abb72(8)
      abb72(7)=abb72(16)+abb72(7)-abb72(8)
      abb72(8)=abb72(7)*mH**2
      abb72(14)=abb72(13)-abb72(14)
      abb72(7)=-abb72(14)*abb72(7)
      abb72(7)=2.0_ki*abb72(7)-abb72(8)
      abb72(7)=2.0_ki*abb72(7)
      abb72(14)=abb72(11)*abb72(2)
      abb72(9)=abb72(14)-abb72(9)
      abb72(9)=mT*abb72(4)*abb72(9)
      abb72(11)=-abb72(13)*abb72(11)*NC
      abb72(9)=abb72(9)+abb72(11)
      abb72(9)=abb72(6)*abb72(9)
      abb72(8)=spbl4k2*abb72(5)*abb72(8)
      abb72(8)=2.0_ki*abb72(9)+abb72(8)
      abb72(8)=2.0_ki*abb72(8)
      R2d72=abb72(12)
      rat2 = rat2 + R2d72
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='72' value='", &
          & R2d72, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd72h14
