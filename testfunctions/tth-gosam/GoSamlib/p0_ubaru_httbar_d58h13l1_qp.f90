module     p0_ubaru_httbar_d58h13l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d58h13l1_qp.f90
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
      use p0_ubaru_httbar_abbrevd58h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc58(25)
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspvak2l3
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      acc58(1)=abb58(9)
      acc58(2)=abb58(10)
      acc58(3)=abb58(11)
      acc58(4)=abb58(12)
      acc58(5)=abb58(13)
      acc58(6)=abb58(14)
      acc58(7)=abb58(15)
      acc58(8)=abb58(16)
      acc58(9)=abb58(17)
      acc58(10)=abb58(19)
      acc58(11)=abb58(20)
      acc58(12)=abb58(21)
      acc58(13)=abb58(22)
      acc58(14)=abb58(23)
      acc58(15)=abb58(29)
      acc58(16)=acc58(7)*Qspvak1l4
      acc58(17)=acc58(11)*Qspvak1l5
      acc58(16)=acc58(17)+acc58(16)+acc58(4)
      acc58(16)=Qspk2*acc58(16)
      acc58(17)=acc58(6)*Qspvak1l4
      acc58(18)=acc58(15)*Qspvak1l5
      acc58(17)=acc58(18)+acc58(9)+acc58(17)
      acc58(17)=QspQ*acc58(17)
      acc58(18)=acc58(2)*Qspvak1l4
      acc58(19)=acc58(12)*Qspvak1l5
      acc58(20)=Qspval5l3*acc58(14)
      acc58(21)=Qspval5k2*acc58(8)
      acc58(22)=Qspval4l3*acc58(13)
      acc58(23)=Qspval4k2*acc58(5)
      acc58(24)=Qspval3l5*acc58(3)
      acc58(25)=Qspvak2l3*acc58(10)
      brack=acc58(1)+acc58(16)+acc58(17)+acc58(18)+acc58(19)+acc58(20)+acc58(21&
      &)+acc58(22)+acc58(23)+acc58(24)+acc58(25)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d58h13l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd58h13_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d58
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d58 = 0.0_ki
      d58 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d58, ki), aimag(d58), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d58h13l1_qp
