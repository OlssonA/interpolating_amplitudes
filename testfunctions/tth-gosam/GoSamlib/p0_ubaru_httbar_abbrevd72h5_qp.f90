module     p0_ubaru_httbar_abbrevd72h5_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh5_qp
   implicit none
   private
   complex(ki), dimension(22), public :: abb72
   complex(ki), public :: R2d72
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_color_qp, only: TR
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      implicit none
      abb72(1)=1.0_ki/(-mT**2+es34)
      abb72(2)=NC**(-1)
      abb72(3)=es12**(-1)
      abb72(4)=spak2l4**(-1)
      abb72(5)=sqrt(mT**2)
      abb72(6)=spak2l3**(-1)
      abb72(7)=spbl3k2**(-1)
      abb72(8)=abb72(2)*c1
      abb72(8)=abb72(8)-c2
      abb72(9)=abb72(8)*abb72(2)
      abb72(10)=c2*NC
      abb72(10)=abb72(10)-c1
      abb72(9)=abb72(9)+abb72(10)
      abb72(11)=abb72(9)*mT
      abb72(12)=abb72(9)*abb72(5)
      abb72(11)=abb72(11)+abb72(12)
      abb72(13)=abb72(11)*spbl4k2
      abb72(14)=abb72(4)*spak2l3
      abb72(15)=-mT*abb72(9)*abb72(14)*spbl3k2
      abb72(13)=abb72(13)-abb72(15)
      abb72(16)=abb72(3)*TR**2*abb72(1)*gs**4*gHT*e*spak1l5*i_
      abb72(17)=2.0_ki*abb72(16)
      abb72(18)=-abb72(13)*abb72(17)
      abb72(19)=spbl4l3*spal3l4
      abb72(20)=mH**2
      abb72(19)=abb72(19)+abb72(20)
      abb72(19)=-abb72(19)*abb72(9)
      abb72(11)=abb72(5)*abb72(11)
      abb72(21)=abb72(20)*abb72(6)*abb72(7)
      abb72(22)=-spbl4k2*abb72(9)*abb72(21)*spak2l4
      abb72(11)=abb72(22)+2.0_ki*abb72(11)+abb72(19)
      abb72(11)=spbl4k2*abb72(5)*abb72(11)
      abb72(19)=abb72(5)**2
      abb72(22)=-abb72(19)*abb72(15)
      abb72(11)=abb72(22)+abb72(11)
      abb72(11)=abb72(11)*abb72(17)
      abb72(12)=-8.0_ki*abb72(12)*abb72(16)*spbl4k2
      abb72(16)=4.0_ki*abb72(16)
      abb72(15)=-abb72(15)*abb72(16)
      abb72(19)=-2.0_ki*abb72(19)+abb72(20)
      abb72(8)=abb72(2)*abb72(4)*abb72(8)
      abb72(10)=abb72(4)*abb72(10)
      abb72(8)=abb72(8)+abb72(10)
      abb72(10)=mT*abb72(8)*abb72(19)
      abb72(19)=-mT-abb72(5)
      abb72(19)=-spbl4k2*abb72(19)*abb72(21)*abb72(9)
      abb72(10)=abb72(19)+abb72(10)
      abb72(10)=abb72(10)*abb72(16)
      abb72(13)=abb72(13)*abb72(16)
      abb72(16)=mT*abb72(17)*spbl4k2
      abb72(14)=abb72(16)*abb72(14)*abb72(9)
      abb72(17)=abb72(17)*abb72(5)
      abb72(19)=-abb72(17)*spbl3k2*abb72(9)
      abb72(9)=abb72(17)*spbl4l3*abb72(9)
      abb72(8)=abb72(16)*spal3l4*abb72(8)
      R2d72=abb72(18)
      rat2 = rat2 + R2d72
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='72' value='", &
          & R2d72, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd72h5_qp
