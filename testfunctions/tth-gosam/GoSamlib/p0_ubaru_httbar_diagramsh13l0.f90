module     p0_ubaru_httbar_diagramsh13l0
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13diagramsl0.f90
   ! generator: buildfortranborn.py
   use p0_ubaru_httbar_color, only: numcs
   use p0_ubaru_httbar_config, only: ki
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   complex(ki), dimension(numcs), parameter :: zero_col = 0.0_ki
   public :: amplitude
contains
!---#[ function amplitude:
   function amplitude()
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_config, only: debug_lo_diagrams, &
        & use_sorted_sum
      use p0_ubaru_httbar_accu, only: sorted_sum
      use p0_ubaru_httbar_util, only: inspect_lo_diagram
      implicit none
      complex(ki), dimension(numcs) :: amplitude
      complex(ki), dimension(14) :: abb
!      complex(ki), dimension(2,numcs) :: diagrams
      integer :: i
      amplitude(:) = 0.0_ki
      abb(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb(2)=es12**(-1)
      abb(3)=spak2l3**(-1)
      abb(4)=spbl3k2**(-1)
      abb(5)=spak2l4**(-1)
      abb(6)=spak2l5**(-1)
      abb(7)=sqrt(mT**2)
      abb(8)=1.0_ki/(-mT**2+es34)
      abb(9)=NC**(-1)
      abb(10)=abb(8)+abb(1)
      abb(11)=i_*TR*e*gHT*abb(2)*gs**2
      abb(10)=abb(11)*spak1k2*abb(10)
      abb(12)=abb(5)*spbl5k2
      abb(13)=abb(6)*spbl4k2
      abb(12)=abb(12)+abb(13)
      abb(13)=abb(7)*abb(12)*mT*abb(10)
      abb(14)=abb(5)*abb(6)*spak2l3*spbl3k2
      abb(12)=abb(14)+abb(12)
      abb(12)=abb(12)*abb(10)*mT**2
      abb(10)=spbl5k2*abb(10)*abb(3)*abb(4)*mH**2
      abb(11)=abb(11)*spak1l3
      abb(14)=abb(1)*abb(11)*spbl5l3
      abb(10)=abb(10)+abb(14)
      abb(10)=abb(10)*spbl4k2
      abb(11)=abb(8)*spbl5k2*abb(11)*spbl4l3
      abb(10)=abb(13)+abb(10)+abb(12)+abb(11)
      abb(11)=2.0_ki*abb(10)
      abb(10)=-2.0_ki*abb(9)*abb(10)
      amplitude=c2*abb(11)+c1*abb(10)
      if (debug_lo_diagrams) then
         write(*,*) "Using Born optimization, debug_lo_diagrams not implemented&
         &."
      end if
!      if (use_sorted_sum) then
!         do i=1,numcs
!            amplitude(i) = sorted_sum(diagrams(i))
!         end do
!      else
!         do i=1,numcs
!            amplitude(i) = sum(diagrams(i))
!         end do
!      end if
   end function     amplitude
!---#] function amplitude:
end module p0_ubaru_httbar_diagramsh13l0
