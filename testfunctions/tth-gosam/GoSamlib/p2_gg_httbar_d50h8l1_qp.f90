module     p2_gg_httbar_d50h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d50h8l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd50h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc50(32)
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval4k2
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspk2 = dotproduct(Q,k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval4k2 = dotproduct(Q,spval4k2)
      acc50(1)=abb50(9)
      acc50(2)=abb50(10)
      acc50(3)=abb50(11)
      acc50(4)=abb50(12)
      acc50(5)=abb50(13)
      acc50(6)=abb50(14)
      acc50(7)=abb50(15)
      acc50(8)=abb50(17)
      acc50(9)=abb50(18)
      acc50(10)=abb50(19)
      acc50(11)=abb50(20)
      acc50(12)=abb50(21)
      acc50(13)=abb50(22)
      acc50(14)=abb50(23)
      acc50(15)=abb50(24)
      acc50(16)=abb50(25)
      acc50(17)=abb50(26)
      acc50(18)=abb50(29)
      acc50(19)=abb50(32)
      acc50(20)=abb50(37)
      acc50(21)=abb50(39)
      acc50(22)=Qspvak2l3*acc50(15)
      acc50(23)=Qspval3k2*acc50(9)
      acc50(24)=Qspk2*acc50(2)
      acc50(22)=acc50(24)+acc50(23)+acc50(5)+acc50(22)
      acc50(22)=Qspk2*acc50(22)
      acc50(23)=-Qspvak1k2*acc50(1)
      acc50(24)=-Qspvak1l3*acc50(19)
      acc50(23)=acc50(24)+acc50(16)+acc50(23)
      acc50(23)=Qspvak2k1*acc50(23)
      acc50(24)=-Qspvak1k2*acc50(21)
      acc50(25)=Qspvak1l5*acc50(20)
      acc50(24)=acc50(25)+acc50(6)+acc50(24)
      acc50(24)=Qspval3k1*acc50(24)
      acc50(25)=Qspvak1l3*acc50(18)
      acc50(26)=Qspvak1l5*acc50(8)
      acc50(25)=acc50(26)+acc50(4)+acc50(25)
      acc50(25)=Qspval4k1*acc50(25)
      acc50(26)=-Qspvak2l3*acc50(18)
      acc50(27)=-Qspvak2l5*acc50(8)
      acc50(26)=acc50(27)+acc50(3)+acc50(26)
      acc50(26)=Qspval4k2*acc50(26)
      acc50(27)=Qspvak1k2*acc50(17)
      acc50(28)=Qspvak1l3*acc50(14)
      acc50(29)=Qspvak1l5*acc50(12)
      acc50(30)=Qspvak2l3*acc50(10)
      acc50(31)=Qspvak2l5*acc50(11)
      acc50(32)=-Qspvak2l5*acc50(20)
      acc50(32)=acc50(7)+acc50(32)
      acc50(32)=Qspval3k2*acc50(32)
      brack=acc50(13)+acc50(22)+acc50(23)+acc50(24)+acc50(25)+acc50(26)+acc50(2&
      &7)+acc50(28)+acc50(29)+acc50(30)+acc50(31)+acc50(32)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d50h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd50h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d50
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k3-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d50 = 0.0_ki
      d50 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d50, ki), aimag(d50), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d50h8l1_qp
