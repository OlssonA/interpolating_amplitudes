module     p0_ubaru_httbar_d64h9l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity9d64h9l1_qp.f90
   ! generator: buildfortran.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd64h9_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc64(24)
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc64(1)=abb64(7)
      acc64(2)=abb64(8)
      acc64(3)=abb64(9)
      acc64(4)=abb64(10)
      acc64(5)=abb64(11)
      acc64(6)=abb64(12)
      acc64(7)=abb64(13)
      acc64(8)=abb64(15)
      acc64(9)=abb64(17)
      acc64(10)=abb64(18)
      acc64(11)=abb64(20)
      acc64(12)=abb64(22)
      acc64(13)=abb64(23)
      acc64(14)=acc64(2)*Qspval3l5
      acc64(15)=acc64(5)*Qspval4l5
      acc64(14)=acc64(8)+acc64(14)+acc64(15)
      acc64(14)=Qspvak1k2*acc64(14)
      acc64(15)=acc64(12)*Qspval3l5
      acc64(16)=acc64(13)*Qspval4l5
      acc64(17)=Qspval5l3*acc64(9)
      acc64(18)=Qspval5k2*acc64(10)
      acc64(19)=Qspval4k2*acc64(1)
      acc64(20)=Qspval3k2*acc64(11)
      acc64(21)=Qspvak1l5*acc64(4)
      acc64(22)=Qspvak1l3*acc64(3)
      acc64(23)=Qspk2*acc64(7)
      acc64(24)=QspQ*acc64(6)
      brack=acc64(14)+acc64(15)+acc64(16)+acc64(17)+acc64(18)+acc64(19)+acc64(2&
      &0)+acc64(21)+acc64(22)+acc64(23)+acc64(24)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d64h9l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd64h9_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d64
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d64 = 0.0_ki
      d64 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d64, ki), aimag(d64), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d64h9l1_qp
