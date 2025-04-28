module     p0_ubaru_httbar_d71h13l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d71h13l1_qp.f90
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
      use p0_ubaru_httbar_abbrevd71h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc71(23)
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl5
      complex(ki) :: Qspl3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl5 = dotproduct(Q,l5)
      Qspl3 = dotproduct(Q,l3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc71(1)=abb71(10)
      acc71(2)=abb71(11)
      acc71(3)=abb71(12)
      acc71(4)=abb71(13)
      acc71(5)=abb71(14)
      acc71(6)=abb71(16)
      acc71(7)=abb71(17)
      acc71(8)=abb71(18)
      acc71(9)=abb71(19)
      acc71(10)=abb71(20)
      acc71(11)=abb71(21)
      acc71(12)=abb71(23)
      acc71(13)=Qspval5l3*acc71(8)
      acc71(14)=Qspval3l5*acc71(9)
      acc71(15)=Qspval3k2*acc71(3)
      acc71(16)=Qspvak2l5*acc71(6)
      acc71(17)=Qspvak2l3*acc71(5)
      acc71(18)=Qspvak1l5*acc71(10)
      acc71(19)=Qspvak1l3*acc71(1)
      acc71(20)=Qspl5*acc71(11)
      acc71(21)=Qspl3*acc71(12)
      acc71(22)=Qspk2*acc71(2)
      acc71(23)=QspQ*acc71(4)
      brack=acc71(7)+acc71(13)+acc71(14)+acc71(15)+acc71(16)+acc71(17)+acc71(18&
      &)+acc71(19)+acc71(20)+acc71(21)+acc71(22)+acc71(23)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d71h13l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd71h13_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d71
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d71 = 0.0_ki
      d71 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d71, ki), aimag(d71), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d71h13l1_qp
