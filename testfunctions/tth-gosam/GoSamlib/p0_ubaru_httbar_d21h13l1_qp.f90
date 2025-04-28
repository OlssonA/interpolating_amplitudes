module     p0_ubaru_httbar_d21h13l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d21h13l1_qp.f90
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
      use p0_ubaru_httbar_abbrevd21h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc21(21)
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl4
      complex(ki) :: Qspl3
      complex(ki) :: Qspk2
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl4 = dotproduct(Q,l4)
      Qspl3 = dotproduct(Q,l3)
      Qspk2 = dotproduct(Q,k2)
      acc21(1)=abb21(12)
      acc21(2)=abb21(13)
      acc21(3)=abb21(14)
      acc21(4)=abb21(15)
      acc21(5)=abb21(16)
      acc21(6)=abb21(18)
      acc21(7)=abb21(19)
      acc21(8)=abb21(20)
      acc21(9)=abb21(21)
      acc21(10)=abb21(22)
      acc21(11)=abb21(23)
      acc21(12)=Qspval4l3*acc21(2)
      acc21(13)=Qspval3l4*acc21(6)
      acc21(14)=Qspval3k2*acc21(3)
      acc21(15)=Qspvak2l4*acc21(9)
      acc21(16)=Qspvak2l3*acc21(8)
      acc21(17)=Qspvak1l4*acc21(11)
      acc21(18)=Qspvak1l3*acc21(10)
      acc21(19)=Qspl4*acc21(7)
      acc21(20)=Qspl3*acc21(4)
      acc21(21)=Qspk2*acc21(1)
      brack=acc21(5)+acc21(12)+acc21(13)+acc21(14)+acc21(15)+acc21(16)+acc21(17&
      &)+acc21(18)+acc21(19)+acc21(20)+acc21(21)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d21h13l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd21h13_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d21
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k4+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d21 = 0.0_ki
      d21 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d21, ki), aimag(d21), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d21h13l1_qp
