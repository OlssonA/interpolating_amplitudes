module     p0_ubaru_httbar_d66h10l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d66h10l1_qp.f90
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
      use p0_ubaru_httbar_abbrevd66h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc66(24)
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc66(1)=abb66(7)
      acc66(2)=abb66(8)
      acc66(3)=abb66(9)
      acc66(4)=abb66(10)
      acc66(5)=abb66(11)
      acc66(6)=abb66(12)
      acc66(7)=abb66(13)
      acc66(8)=abb66(14)
      acc66(9)=abb66(15)
      acc66(10)=abb66(16)
      acc66(11)=abb66(17)
      acc66(12)=abb66(20)
      acc66(13)=abb66(21)
      acc66(14)=acc66(6)*Qspval4l3
      acc66(15)=acc66(7)*Qspval4l5
      acc66(14)=acc66(15)+acc66(5)+acc66(14)
      acc66(14)=Qspvak2k1*acc66(14)
      acc66(15)=acc66(12)*Qspval4l3
      acc66(16)=acc66(13)*Qspval4l5
      acc66(17)=Qspval4k1*acc66(1)
      acc66(18)=Qspval3l4*acc66(9)
      acc66(19)=Qspval3k1*acc66(3)
      acc66(20)=Qspvak2l5*acc66(10)
      acc66(21)=Qspvak2l4*acc66(11)
      acc66(22)=Qspvak2l3*acc66(8)
      acc66(23)=Qspk2*acc66(2)
      acc66(24)=QspQ*acc66(4)
      brack=acc66(14)+acc66(15)+acc66(16)+acc66(17)+acc66(18)+acc66(19)+acc66(2&
      &0)+acc66(21)+acc66(22)+acc66(23)+acc66(24)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d66h10l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd66h10_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d66
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d66 = 0.0_ki
      d66 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d66, ki), aimag(d66), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d66h10l1_qp
