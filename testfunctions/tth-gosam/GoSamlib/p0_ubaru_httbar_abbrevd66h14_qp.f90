module     p0_ubaru_httbar_abbrevd66h14_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh14_qp
   implicit none
   private
   complex(ki), dimension(24), public :: abb66
   complex(ki), public :: R2d66
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
      abb66(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb66(2)=NC**(-1)
      abb66(3)=spak2l4**(-1)
      abb66(4)=spbl4k2**(-1)
      abb66(5)=sqrt(mT**2)
      abb66(6)=spak2l5**(-1)
      abb66(7)=abb66(2)*c1
      abb66(8)=abb66(7)*spbl5l3
      abb66(9)=c2*spbl5l3
      abb66(8)=-abb66(8)+2.0_ki*abb66(9)
      abb66(8)=abb66(8)*abb66(2)
      abb66(9)=abb66(9)*NC
      abb66(8)=abb66(8)-abb66(9)
      abb66(9)=spak2l3*abb66(3)
      abb66(10)=mT**2
      abb66(11)=abb66(9)*abb66(10)
      abb66(12)=abb66(1)*gHT*e*i_*gs**4*TR**2
      abb66(13)=abb66(12)*spbl4k1
      abb66(13)=4.0_ki*abb66(13)
      abb66(14)=-abb66(13)*abb66(8)*abb66(11)*abb66(4)
      abb66(13)=-abb66(13)*spak2l3*abb66(8)
      abb66(15)=2.0_ki*spbl4k1
      abb66(15)=abb66(15)*abb66(12)
      abb66(16)=-abb66(15)*spak2l4*abb66(8)
      abb66(17)=-abb66(15)*spak1k2*abb66(8)
      abb66(18)=2.0_ki*c2
      abb66(7)=abb66(7)-abb66(18)
      abb66(19)=abb66(5)**2
      abb66(10)=abb66(19)-abb66(10)
      abb66(19)=-abb66(2)*abb66(10)*abb66(7)
      abb66(20)=c2*NC
      abb66(10)=-abb66(10)*abb66(20)
      abb66(10)=abb66(10)+abb66(19)
      abb66(10)=abb66(10)*abb66(15)
      abb66(19)=abb66(15)*spal3l4*abb66(8)
      abb66(21)=NC*c2*mT
      abb66(22)=abb66(5)-mT
      abb66(22)=abb66(22)*spak2l3*abb66(6)
      abb66(23)=-abb66(22)*abb66(21)
      abb66(24)=c1*abb66(2)*mT
      abb66(18)=abb66(18)*mT
      abb66(18)=abb66(24)-abb66(18)
      abb66(22)=-abb66(2)*abb66(22)*abb66(18)
      abb66(22)=abb66(23)+abb66(22)
      abb66(15)=abb66(22)*abb66(15)
      abb66(9)=abb66(8)*abb66(5)*abb66(9)*mT
      abb66(8)=spbl4k1*spak1l3*abb66(8)
      abb66(8)=abb66(8)+abb66(9)
      abb66(9)=2.0_ki*abb66(12)
      abb66(8)=abb66(8)*abb66(9)
      abb66(12)=abb66(5)+mT
      abb66(12)=abb66(3)*abb66(12)
      abb66(21)=abb66(12)*abb66(21)
      abb66(12)=abb66(2)*abb66(12)*abb66(18)
      abb66(12)=abb66(21)+abb66(12)
      abb66(12)=abb66(12)*abb66(9)
      abb66(7)=abb66(2)*abb66(7)
      abb66(7)=abb66(20)+abb66(7)
      abb66(7)=abb66(9)*abb66(7)*abb66(11)*abb66(6)
      R2d66=0.0_ki
      rat2 = rat2 + R2d66
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='66' value='", &
          & R2d66, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd66h14_qp
